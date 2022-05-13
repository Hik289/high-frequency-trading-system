package com.boot.entity;

import java.io.Serializable;
import java.util.Date;
import java.util.List;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.Table;
import javax.persistence.Transient;

import lombok.Data;

/**
 * board list
 */
@Entity
@Table(name = "t_board_list")
@Data
public class BoardList implements Serializable{
	private static final long serialVersionUID = 2527440173722512927L;
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private Integer id;
	private String name;
	
	@Column(name="board_id")
	private Integer boardId;//
	
	private Integer type = 1;//0 系统自带，1 用户创建
	
	@Column(name="create_time",updatable=false)
	private Date createTime = new Date();//创建时间
	
	@Transient
	private List<BoardListCard> listCard;
}
